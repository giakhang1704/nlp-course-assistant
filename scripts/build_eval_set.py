"""Build the final evaluation set (Part IV): 34 official queries + 47 group queries,
5 official dialogues + 3 group dialogues.

Official items come from NLP_Course_Assistant_Sample_Data (sample_queries.txt,
dialogues.txt); expected intent/entity are the lecturer's labels. Group items reuse
gold already written in data/evaluation/questions.jsonl where possible.
Gold evidence is the KB document id(s) that must support the answer; an inner list
of evidence_groups lists alternatives (any one of them is enough).

Run from the project root:  python -B scripts/build_eval_set.py
Writes data/evaluation/final/{parser_cases,qa_cases,dialogues}.jsonl,
input/sentences_eval.txt (84 lines, parser) and input/questions_eval.txt (81 lines, QA;
the 3 malformed sentences are parser-only).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "evaluation" / "final"
OLD = {json.loads(l)["id"]: json.loads(l) for l in
       (ROOT / "data/evaluation/questions.jsonl").read_text(encoding="utf-8").splitlines() if l}

ANS, MISS, CLAR = "retrieved", "insufficient_information", "clarification_needed"


def case(cid, source, category, question, status, groups=(), answer=None, facts=None,
         parsed=None, intent=None, entities=(), paraphrase=None, qa=True, note=None):
    groups = [list(g) for g in groups]
    return {"id": cid, "source": source, "category": category, "question": question,
            "expected_status": status, "evidence_groups": groups,
            "relevant_documents": sorted({d for g in groups for d in g}),
            "reference_answer": answer, "required_facts": facts or ([answer] if answer else []),
            "parser_expected": parsed, "intent_expected": intent, "entities_expected": list(entities),
            "paraphrase_group": paraphrase, "qa_evaluated": qa, "note": note}


def reuse(cid, old_id, category, intent=None, entities=(), paraphrase=None, qa=True):
    o = OLD[old_id]
    sem = o.get("semantic_expected") or {}
    status = o["expected_status"]
    return case(cid, f"group (questions.jsonl {old_id})", category, o["question"], status,
                o.get("evidence_groups") or [], o.get("reference_answer"), o.get("required_facts"),
                o.get("parser_expected"), intent or sem.get("intent"), entities or sem.get("entity_ids") or (),
                paraphrase, qa and o["expected_status"] is not None)


# ---- 34 official queries (sample_queries.txt) ----------------------------------------
O = "official"
official = [
    case("O01", O, "answerable", "Môn NLP có bao nhiêu tín chỉ?", ANS, [["course_info.credits"]], "Môn học có 3 tín chỉ.", parsed=True, intent="GET_COURSE_INFO", entities=["CO3085"]),
    case("O02", O, "answerable", "CO3085 có mấy tín chỉ?", ANS, [["course_info.credits"]], "Môn học có 3 tín chỉ.", parsed=True, intent="GET_COURSE_INFO", entities=["CO3085"]),
    case("O03", O, "answerable", "Môn học gồm những phần nào?", ANS, [["faq.q02"]], "11 chương, chia thành Part I Syntax Processing (chương 2–5), Part II Semantic Interpretation (chương 7–10), Part III Context and Knowledge Representation (chương 11), cùng chương 1 tổng quan và chương 6 phương pháp thống kê.", parsed=True, intent="GET_COURSE_INFO", entities=["CO3085"]),
    case("O04", O, "answerable", "Tuần 3 học gì?", ANS, [["schedule.week_03"]], "Tuần 3 học Chương 3: CFG, top-down/bottom-up chart parsing, PCFG, dependency grammar.", parsed=True, intent="GET_SCHEDULE", entities=["WEEK_03"]),
    case("O05", O, "answerable", "Chương 4 học vào tuần nào?", ANS, [["schedule.week_04"], ["schedule.week_05"]], "Chương 4 học vào tuần 4 và tuần 5.", parsed=True, intent="GET_SCHEDULE", entities=["CH04"]),
    case("O06", O, "answerable", "Chương nào nói về CFG?", ANS, [["topics.chapter_03", "faq.q04", "schedule.week_03"]], "CFG thuộc Chương 3.", parsed=True, intent="GET_TOPIC", entities=["CFG"]),
    case("O07", O, "answerable", "PCFG được học ở đâu?", ANS, [["schedule.week_03", "topics.chapter_03", "slide_ch03_pcfg"]], "PCFG được học ở Chương 3 (tuần 3).", parsed=True, intent="GET_TOPIC", entities=["PCFG"]),
    case("O08", O, "answerable", "Chương nào nói về semantic grammar?", ANS, [["topics.chapter_10", "faq.q13", "slide_ch10_semantic_grammar"]], "Semantic grammar thuộc Chương 10.", parsed=True, intent="GET_TOPIC", entities=["SEMANTIC_GRAMMAR"]),
    case("O09", O, "answerable", "L.O.2.5 là gì?", ANS, [["course_info.detailed_learning_outcomes"]], "LO2.5 là kiến thức xác suất để giải quyết nhập nhằng từ ở mức câu.", parsed=True, intent="GET_LO", entities=["LO2.5"]),
    case("O10", O, "answerable", "L.O.4.2 liên quan đến nội dung nào?", ANS, [["course_info.detailed_learning_outcomes"]], "LO4.2 liên quan đến ngữ cảnh diễn ngôn cục bộ và sự tham chiếu.", parsed=True, intent="GET_LO", entities=["LO4.2"]),
    case("O11", O, "answerable", "Bài tập lớn yêu cầu xây dựng những gì?", ANS, [["assignments.overview"], ["assignments.part_1", "assignments.part_2"]], "Xây dựng hệ thống Course Assistant thể hiện pipeline NLP từ parsing đến semantic interpretation và hỏi đáp (grammar/parser, semantic, QA...).", parsed=True, intent="GET_ASSIGNMENT", entities=["BTL01"]),
    case("O12", O, "answerable", "Phần I của bài tập lớn là gì?", ANS, [["assignments.part_1"]], "Phần I là Grammar and Parser: xây dựng grammar, sinh câu, xây dựng parser, xuất parse tree, xử lý câu không hợp lệ.", parsed=True, intent="GET_ASSIGNMENT", entities=["PART_I"]),
    case("O13", O, "answerable", "Bài tập lớn có phần hỏi đáp không?", ANS, [["assignments.part_2"]], "Có, Phần II là Semantic Interpretation and QA.", parsed=True, intent="GET_ASSIGNMENT", entities=["PART_II"]),
    case("O14", O, "answerable", "Có được dùng LLM/RAG không?", ANS, [["regulations.llm_rag", "assignments.part_5"]], "Có thể dùng như phần mở rộng; phải mô tả rõ mô hình, dữ liệu, retrieval và cách đánh giá.", parsed=True, intent="GET_RULE", entities=["LLM_RAG"]),
    case("O15", O, "answerable", "Quy định về sử dụng AI là gì?", ANS, [["regulations.ai_usage"]], "Sinh viên phải khai báo việc sử dụng AI trong báo cáo, nghiên cứu và quá trình thực hiện.", parsed=True, intent="GET_RULE", entities=["AI_USAGE"]),
    case("O16", O, "answerable", "Midterm bao nhiêu phút?", ANS, [["course_info.midterm"]], "Midterm 60 phút.", parsed=True, intent="GET_COURSE_INFO", entities=["MIDTERM"]),
    case("O17", O, "answerable", "Final bao nhiêu phút?", ANS, [["course_info.final"]], "Final 90 phút.", parsed=True, intent="GET_COURSE_INFO", entities=["FINAL"]),
    case("O18", O, "answerable", "Tài liệu tham khảo về statistical NLP là gì?", ANS, [["resources.reference_1"]], "Foundation of Statistical Natural Language Processing (Manning, Schütze, 2001).", parsed=True, intent="GET_RESOURCE", entities=["STATISTICAL_NLP"]),
    case("O19", O, "answerable", "Tài liệu nào liên quan đến Python NLP?", ANS, [["resources.reference_2"]], "Natural Language Processing with Python (Bird, Klein, Loper, 2009).", parsed=True, intent="GET_RESOURCE", entities=["PYTHON_NLP"]),
    case("O20", O, "answerable", "Chương 12 nói về gì?", ANS, [["topics.chapter_12", "schedule.week_14"]], "Chương 12 nói về Discourse Context and Reference: local discourse context, anaphora, pronouns/centering, definite descriptions, ellipsis.", parsed=True, intent="GET_TOPIC", entities=["CH12"]),
    case("O21", O, "answerable", "Chương nào học về reference?", ANS, [["topics.chapter_12", "schedule.week_14"]], "Chương 12 (Discourse Context and Reference).", parsed=True, intent="GET_TOPIC", entities=["REFERENCE"]),
    case("O22", O, "answerable", "Chương nào học về question answering?", ANS, [["slide_ch01_question_answering", "topics.chapter_01"]], "Question answering được giới thiệu ở Chương 1; Chương 11 có procedural semantics cho hỏi đáp cơ sở dữ liệu.", parsed=True, intent="GET_TOPIC", entities=["QUESTION_ANSWERING"]),
    case("O23", O, "answerable", "Chương nào học về selectional restriction?", ANS, [["topics.chapter_09", "faq.q12", "slide_ch09_selectional_restrictions"]], "Selectional restriction thuộc Chương 9.", parsed=True, intent="GET_TOPIC", entities=["SELECTIONAL_RESTRICTION"]),
    case("O24", O, "answerable", "Chương 10 có những phương pháp gì?", ANS, [["topics.chapter_10", "schedule.week_12"]], "Grammatical relations (LSUBJ/LOBJ), pattern matching, semantic grammars.", parsed=True, intent="GET_TOPIC", entities=["CH10"]),
    case("O25", O, "answerable", "Chương 11 có liên quan đến hỏi đáp không?", ANS, [["topics.chapter_11", "slide_ch11_procedural_semantics", "faq.q15"]], "Có, Chương 11 có procedural semantics dùng cho hỏi đáp/truy vấn cơ sở dữ liệu.", parsed=True, intent="GET_TOPIC", entities=["CH11"]),
    case("O26", O, "out_of_domain", "Hôm nay ở TP.HCM có mưa không?", MISS, parsed=False, intent="UNKNOWN"),
    case("O27", O, "out_of_domain", "Ai là tổng thống của Pháp?", MISS, parsed=False, intent="UNKNOWN"),
    case("O28", O, "out_of_domain", "Hãy giải bài tập xác suất này giúp tôi.", MISS, parsed=False, intent="UNKNOWN"),
    case("O29", O, "answerable", "Chương 4 nói về gì?", ANS, [["topics.chapter_04"]], "Chương 4 nói về Features và Augmented Grammar.", parsed=True, intent="GET_TOPIC", entities=["CH04"]),
    case("O30", O, "context_dependent", "Nó có liên quan đến parsing không?", CLAR, parsed=True, intent="CONTEXT_DEPENDENT", note="Hỏi riêng lẻ phải hỏi lại; trong hội thoại OD02 thì 'Nó' = CH04."),
    case("O31", O, "answerable", "Chương 11 có hỏi đáp không?", ANS, [["topics.chapter_11", "slide_ch11_procedural_semantics", "faq.q15"]], "Có, Chương 11 có procedural semantics cho hỏi đáp.", parsed=True, intent="GET_TOPIC", entities=["CH11"]),
    case("O32", O, "context_dependent", "Còn chương sau?", CLAR, parsed=True, intent="CONTEXT_DEPENDENT", note="Hỏi riêng lẻ phải hỏi lại; xem hội thoại OD04."),
    case("O33", O, "answerable", "Bài tập lớn có phần parser không?", ANS, [["assignments.part_1"]], "Có, Phần I yêu cầu xây dựng grammar và parser.", parsed=True, intent="GET_ASSIGNMENT", entities=["PART_I"]),
    case("O34", O, "context_dependent", "Phần này cần xuất file gì?", CLAR, parsed=True, intent="CONTEXT_DEPENDENT", note="Hỏi riêng lẻ phải hỏi lại; xem hội thoại OD03."),
]

# ---- 47 group queries -------------------------------------------------------------------
G = "group"
group = [
    # A. parsed by the grammar, answerable, one per query type (WH / yes-no / request)
    reuse("N01", "Q003", "answerable"),
    reuse("N02", "Q008", "answerable"),
    reuse("N03", "Q009", "answerable"),
    reuse("N04", "Q011", "answerable"),
    reuse("N05", "Q015", "answerable"),
    reuse("N06", "Q019", "answerable"),
    reuse("N07", "Q020", "answerable"),
    reuse("N08", "Q021", "answerable"),
    reuse("N09", "Q027", "answerable"),
    reuse("N10", "Q028", "answerable"),
    reuse("N11", "Q024", "answerable"),
    reuse("N12", "Q030", "answerable"),
    case("N13", G, "answerable", "Liệt kê tài liệu tham khảo.", ANS,
         [["resources.reference_1", "resources.reference_2", "resources.reference_3", "resources.reference_4", "resources.reference_5", "faq.q20"]],
         "Liệt kê được các tài liệu tham khảo trong KB, ví dụ Speech and Language Processing (Jurafsky, Martin), Foundation of Statistical NLP (Manning, Schütze), NLP with Python (Bird et al.).",
         parsed=False, intent="GET_RESOURCE"),
    reuse("N14", "Q017", "missing_information"),
    # B. paraphrase groups: same request, different wording (WH / request / natural)
    case("N15", G, "answerable", "Bài tập lớn chiếm bao nhiêu phần trăm?", ANS, [["course_info.major_assignment"]], "Bài tập lớn chiếm 15%.", parsed=False, intent="GET_COURSE_INFO", paraphrase="P1_weight"),
    case("N16", G, "answerable", "Trọng số của bài tập lớn là bao nhiêu?", ANS, [["course_info.major_assignment"]], "Bài tập lớn chiếm 15%.", parsed=False, intent="GET_COURSE_INFO", paraphrase="P1_weight"),
    case("N18", G, "answerable", "CFG là gì?", ANS, [["slide_ch03_cfg"]], "CFG gồm start symbol, productions, nonterminals và terminals; mỗi production có dạng A → α với A là nonterminal.", parsed=True, intent="GET_TOPIC", entities=["CFG"], paraphrase="P2_cfg"),
    case("N19", G, "answerable", "Giải thích CFG.", ANS, [["slide_ch03_cfg"]], "CFG gồm start symbol, productions, nonterminals và terminals; mỗi production có dạng A → α với A là nonterminal.", parsed=True, intent="GET_TOPIC", entities=["CFG"], paraphrase="P2_cfg"),
    case("N20", G, "answerable", "Context-Free Grammar được định nghĩa như thế nào?", ANS, [["slide_ch03_cfg"]], "CFG gồm start symbol, productions, nonterminals và terminals; mỗi production có dạng A → α với A là nonterminal.", parsed=True, intent="GET_TOPIC", entities=["CFG"], paraphrase="P2_cfg"),
    case("N21", G, "answerable", "Tuần 5 học những gì?", ANS, [["schedule.week_05"]], "Tuần 5 (Chương 4): augmented transition networks, generalized feature systems, unification grammar; bài tập feature-based parsing.", parsed=True, intent="GET_SCHEDULE", entities=["WEEK_05"], paraphrase="P3_week5"),
    case("N22", G, "answerable", "Nội dung của tuần 5.", ANS, [["schedule.week_05"]], "Tuần 5 (Chương 4): augmented transition networks, generalized feature systems, unification grammar; bài tập feature-based parsing.", parsed=True, intent="GET_SCHEDULE", entities=["WEEK_05"], paraphrase="P3_week5"),
    case("N23", G, "answerable", "Tuần thứ năm của môn có những chủ đề nào?", ANS, [["schedule.week_05"]], "Tuần 5 (Chương 4): augmented transition networks, generalized feature systems, unification grammar; bài tập feature-based parsing.", parsed=False, intent="GET_SCHEDULE", entities=["WEEK_05"], paraphrase="P3_week5"),
    case("N25", G, "answerable", "Nêu quy định về việc dùng AI.", ANS, [["regulations.ai_usage"]], "Sinh viên phải khai báo việc sử dụng AI trong báo cáo, nghiên cứu và quá trình thực hiện.", parsed=False, intent="GET_RULE", paraphrase="P4_ai"),
    case("N26", G, "answerable", "Sinh viên dùng AI thì phải làm gì?", ANS, [["regulations.ai_usage"]], "Sinh viên phải khai báo việc sử dụng AI trong báo cáo, nghiên cứu và quá trình thực hiện.", parsed=False, intent="GET_RULE", paraphrase="P4_ai"),
    case("N27", G, "answerable", "Phần II của bài tập lớn yêu cầu gì?", ANS, [["assignments.part_2"]], "Semantic representation, intent identification, entity extraction, query tới KB, sinh câu trả lời, xử lý câu ngoài miền và trường hợp không tìm thấy thông tin.", parsed=True, intent="GET_ASSIGNMENT", entities=["PART_II"], paraphrase="P5_part2"),
    case("N28", G, "answerable", "Liệt kê yêu cầu của phần hỏi đáp.", ANS, [["assignments.part_2"]], "Semantic representation, intent identification, entity extraction, query tới KB, sinh câu trả lời, xử lý câu ngoài miền và trường hợp không tìm thấy thông tin.", parsed=True, intent="GET_ASSIGNMENT", entities=["PART_II"], paraphrase="P5_part2"),
    # C. natural questions outside the grammar, answerable (raw_question route)
    reuse("N30", "Q032", "answerable"),
    reuse("N31", "Q035", "answerable"),
    reuse("N32", "Q039", "answerable"),
    reuse("N33", "Q043", "answerable"),
    reuse("N34", "Q055", "answerable"),
    # D. in-domain but the KB has no answer
    reuse("N35", "Q062", "missing_information"),
    reuse("N36", "Q066", "missing_information"),
    reuse("N37", "Q069", "missing_information"),
    reuse("N38", "Q081", "missing_information"),
    case("N39", G, "missing_information", "Email của giảng viên là gì?", MISS, parsed=True, intent="GET_INSTRUCTOR"),
    # E. out of domain
    reuse("N40", "Q071", "out_of_domain"),
    reuse("N41", "Q074", "out_of_domain"),
    case("N42", G, "out_of_domain", "Thời tiết Đà Lạt ngày mai ra sao?", MISS, parsed=False, intent="UNKNOWN"),
    case("N43", G, "out_of_domain", "Viết giúp tôi một bài thơ về mùa thu.", MISS, parsed=False, intent="UNKNOWN"),
    # F. context-dependent asked alone: must ask back, not guess
    case("N44", G, "context_dependent", "Thuật toán này là gì?", CLAR, parsed=True, intent="CONTEXT_DEPENDENT"),
    case("N45", G, "context_dependent", "Tài liệu của phần này ở đâu?", CLAR, parsed=True, intent="CONTEXT_DEPENDENT"),
    case("N46", G, "context_dependent", "Chương đó học vào tuần nào?", CLAR, parsed=False, intent="CONTEXT_DEPENDENT"),
    reuse("N47", "Q088", "context_dependent", intent="CONTEXT_DEPENDENT"),
    # G. malformed: parser must output (), checked in the parser evaluation only
    reuse("N48", "Q091", "malformed", qa=False),
    reuse("N49", "Q094", "malformed", qa=False),
    reuse("N50", "Q097", "malformed", qa=False),
]


# Automatic answer check: every item must appear in the answer (case-insensitive);
# "a|b" lists alternatives. Keywords are the key fact of the reference answer, in
# Vietnamese or the English term kept in the KB.
CH = lambda n: f"chương {n}|chapter {n}|ch{n:02d}"
WK = lambda n: f"tuần {n}|week {n}|week {n:02d}|tuần {n:02d}"
MUST = {
    "O01": ["3"], "O02": ["3"], "O03": ["part i|phần i|syntax", "part ii|phần ii|semantic"],
    "O04": ["context-free grammar|cfg|văn phạm phi ngữ cảnh"], "O05": [WK(4), WK(5)],
    "O06": [CH(3)], "O07": [CH(3) + "|" + WK(3)], "O08": [CH(10)], "O09": ["xác suất|probabilit"],
    "O10": ["diễn ngôn|discourse", "tham chiếu|reference"], "O11": ["grammar|văn phạm|ngữ pháp|parser|parsing"],
    "O12": ["grammar|văn phạm|ngữ pháp", "parser|phân tích cú pháp|bộ phân tích"], "O13": ["phần ii|part ii|semantic|hỏi đáp|qa"],
    "O14": ["mở rộng|extension"], "O15": ["khai báo"], "O16": ["60"], "O17": ["90"],
    "O18": ["manning|foundation of statistical"], "O19": ["python"], "O20": ["discourse|diễn ngôn|ngữ cảnh hội thoại", "reference|tham chiếu|anaphora"], "O21": [CH(12)], "O22": [CH(1)],
    "O23": [CH(9)], "O24": ["grammatical relation|quan hệ văn phạm|semantic grammar|văn phạm ngữ nghĩa|ngữ pháp ngữ nghĩa|pattern"],
    "O25": ["hỏi đáp|question answering|procedural|thủ tục|truy vấn|query"], "O29": ["feature|augmented|đặc trưng|gia tố"],
    "O31": ["hỏi đáp|question answering|procedural|thủ tục|truy vấn|query"], "O33": ["phần i|part i|grammar|parser"],
    "N01": ["150"], "N02": ["30"], "N03": ["logical form|semantics|ngữ nghĩa"],
    "N04": ["lambda|compositional|sem"], "N05": ["grammar.txt", "samples.txt", "parse-results.txt"],
    "N06": ["mở rộng|extension"], "N07": ["cfg|context-free grammar"], "N08": ["quan hệ văn phạm|văn phạm ngữ nghĩa"],
    "N09": ["backpointer|backptr|lần ngược|truy vết|truy ngược"], "N10": ["probabilit|xác suất"],
    "N11": ["gap", "khuyết|missing|thiếu"], "N12": ["log"], "N13": ["jurafsky|manning|python|speech and language"],
    "N15": ["15"], "N16": ["15"],
    "N18": ["production|luật sinh|quy tắc"], "N19": ["production|luật sinh|quy tắc"], "N20": ["production|luật sinh|quy tắc"],
    "N21": ["unification|augmented transition|feature"], "N22": ["unification|augmented transition|feature"],
    "N23": ["unification|augmented transition|feature"],
    "N25": ["khai báo"], "N26": ["khai báo"],
    "N27": ["intent|ý định", "entity|thực thể"], "N28": ["intent|ý định", "entity|thực thể"],
    "N30": ["ontolog|bán cấu trúc|semi-structured"], "N31": ["n³|n^3|n3|bậc ba|bậc 3"],
    "N32": ["finite|đầu tiên|first"], "N33": ["lượng từ|quantifier|scope|phạm vi"],
    "N34": ["open-book|open book|mở tài liệu|được dùng tài liệu|được sử dụng tài liệu"],
}
for _case in official + group:
    _case["must_contain"] = MUST.get(_case["id"], [])


def turn(i, status, targets, focus, answer, must=()):
    return {"turn_index": i, "expected_status": status, "reference_targets": targets,
            "expected_focus_document_ids": focus, "required_facts": [answer] if answer else [],
            "reference_answer": answer, "must_contain": list(must)}


A = "answered"
dialogues = [
    {"session_id": "OD01", "source": "official", "turns": ["Tuần 3 học gì?", "Tài liệu của chương này là gì?"], "gold_turns": [
        turn(1, A, [], ["schedule.week_03"], "Tuần 3 học Chương 3: CFG, chart parsing, PCFG, dependency grammar.", ["context-free grammar|cfg|văn phạm phi ngữ cảnh"]),
        turn(2, A, ["CH03"], ["resources.reference_3", "faq.q20", "faq.q04", "topics.chapter_03", "slide_ch03_cfg", "slide_ch03_grammar_quality", "slide_ch03_top_down", "slide_ch03_chart", "slide_ch03_pcfg", "slide_ch03_dependency", "slide_ch03_projectivity", "slide_ch03_head_rules", "slide_ch03_transition_parsing", "slide_ch03_arc_eager", "slide_ch03_dependency_evaluation"], "Dùng tài liệu tham khảo của môn (ví dụ Speech and Language Processing); 'chương này' = Chương 3.", ["speech and language|jurafsky|slide|tài liệu"])]},
    {"session_id": "OD02", "source": "official", "turns": ["Chương 4 nói về gì?", "Nó có liên quan đến parsing không?"], "gold_turns": [
        turn(1, A, [], ["topics.chapter_04"], "Chương 4 nói về Features và Augmented Grammar.", ["feature|augmented|đặc trưng|gia tố"]),
        turn(2, A, ["CH04"], ["topics.chapter_04", "schedule.week_04"], "Có, Chương 4 có parsing với features/augmented grammar.", ["parsing|phân tích cú pháp"])]},
    {"session_id": "OD03", "source": "official", "turns": ["Bài tập lớn có phần parser không?", "Phần này cần xuất file gì?"], "gold_turns": [
        turn(1, A, [], ["assignments.part_1"], "Có, Phần I yêu cầu xây dựng grammar và parser.", ["phần i|part i|grammar|parser"]),
        turn(2, A, ["PART_I"], ["assignments.part_1"], "grammar.txt, samples.txt, parse-results.txt.", ["grammar.txt", "samples.txt", "parse-results.txt"])]},
    {"session_id": "OD04", "source": "official", "turns": ["Chương 11 có hỏi đáp không?", "Còn chương sau?"], "gold_turns": [
        turn(1, A, [], ["topics.chapter_11", "slide_ch11_procedural_semantics"], "Có, Chương 11 có procedural semantics cho hỏi đáp.", ["hỏi đáp|question answering|procedural|thủ tục|truy vấn|query"]),
        # reference_targets = the antecedent the relative reference starts from
        turn(2, A, ["CH11"], ["topics.chapter_12", "schedule.week_14"], "Chương 12 tập trung vào discourse context và reference.", ["discourse|diễn ngôn|reference|tham chiếu|anaphora"])]},
    {"session_id": "OD05", "source": "official", "turns": ["L.O.2.5 học về gì?", "Nội dung này nằm ở chương nào?"], "gold_turns": [
        turn(1, A, [], ["course_info.detailed_learning_outcomes"], "LO2.5: kiến thức xác suất để giải quyết nhập nhằng từ ở mức câu.", ["xác suất|probabilit"]),
        turn(2, A, ["LO2.5"], ["topics.chapter_06", "faq.q07"], "Chương 6 (Ambiguity Resolution: Statistical Methods).", [CH(6)])]},
    {"session_id": "GD01", "source": "group", "turns": ["Viterbi là gì?", "Thuật toán này dùng ở chương nào?", "Chương đó học vào tuần nào?"], "gold_turns": [
        turn(1, A, [], ["slide_ch06_viterbi_forward"], "Viterbi tìm chuỗi category có score lớn nhất, giữ path tốt nhất và backpointer.", ["path|đường|backpointer|score|chuỗi"]),
        turn(2, A, ["VITERBI"], ["faq.q07", "topics.chapter_06"], "Viterbi thuộc Chương 6.", [CH(6)]),
        turn(3, A, ["CH06"], ["schedule.week_07", "schedule.week_08"], "Chương 6 học ở tuần 7 và tuần 8.", [WK(7), WK(8)])]},
    {"session_id": "GD02", "source": "group", "turns": ["Bài tập lớn gồm những phần nào?", "Phần I cần làm gì?", "Còn Phần II?"], "gold_turns": [
        turn(1, A, [], ["assignments.overview", "assignments.part_1", "assignments.part_2"], "Gồm các phần: Grammar and Parser, Semantic Interpretation and QA, ... (theo assignments).", ["parser|grammar|văn phạm|ngữ pháp"]),
        turn(2, A, [], ["assignments.part_1"], "Xây dựng grammar, sinh câu, xây dựng parser, xuất parse tree, xử lý câu không hợp lệ.", ["grammar|văn phạm|ngữ pháp"]),
        turn(3, A, [], ["assignments.part_2"], "Semantic representation, intent, entity, query KB, sinh câu trả lời, xử lý ngoài miền/không tìm thấy.", ["intent|ý định"])]},
    {"session_id": "GD03", "source": "group", "turns": ["So sánh CFG và PCFG.", "Thời tiết hôm nay thế nào?"], "gold_turns": [
        turn(1, A, [], ["slide_ch03_cfg", "slide_ch03_pcfg"], "PCFG bổ sung xác suất cho các production của CFG.", ["probabilit|xác suất"]),
        turn(2, MISS, [], [], "Ngoài miền; không được dùng ngữ cảnh CFG/PCFG để trả lời.")]},
]



def main():
    cases = official + group
    ids = [c["id"] for c in cases]
    questions = [c["question"] for c in cases]
    assert len(official) == 34 and len(group) == 47
    assert len(set(ids)) == len(ids) and len(set(questions)) == len(questions), "duplicate id/question"
    OUT.mkdir(parents=True, exist_ok=True)
    qa = [c for c in cases if c["qa_evaluated"]]
    (OUT / "parser_cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    (OUT / "qa_cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in qa), encoding="utf-8")
    (OUT / "dialogues.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dialogues), encoding="utf-8")
    (ROOT / "input" / "sentences_eval.txt").write_text("\n".join(questions) + "\n", encoding="utf-8")
    (ROOT / "input" / "questions_eval.txt").write_text("\n".join(c["question"] for c in qa) + "\n", encoding="utf-8")
    from collections import Counter
    print("cases:", len(cases), "qa:", len(qa), dict(Counter(c["category"] for c in cases)))
    print("parser_expected:", dict(Counter(c["parser_expected"] for c in cases)))
    print("paraphrase groups:", dict(Counter(c["paraphrase_group"] for c in cases if c["paraphrase_group"])))
    print("intents:", dict(Counter(c["intent_expected"] for c in cases)))
    print("dialogues:", len(dialogues), "turns:", sum(len(d["turns"]) for d in dialogues))


if __name__ == "__main__":
    main()
